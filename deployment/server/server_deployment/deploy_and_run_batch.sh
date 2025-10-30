#!/bin/bash
# VEC项目 - 批量参数敏感性实验部署脚本
# 用途：部署到AutoDL服务器并运行完整的8个参数对比实验

# ========== 服务器配置 ==========
SERVER_HOST="region-9.autodl.pro"
SERVER_PORT="47042"
SERVER_USER="root"
SERVER_PASSWORD="dfUJkmli0mHk"
REMOTE_DIR="/root/VEC_mig_caching"

echo "=========================================="
echo "VEC批量实验部署脚本"
echo "目标服务器: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT}"
echo "实验模式: full (500轮/配置, 预计2-5天)"
echo "=========================================="

# ========== 步骤1：测试连接 ==========
echo ""
echo "[1/6] 测试服务器连接..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_HOST} "echo '✅ 连接成功！'" || {
    echo "❌ 连接失败！"
    echo ""
    echo "💡 手动连接方法："
    echo "   ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}"
    echo "   密码: ${SERVER_PASSWORD}"
    echo ""
    echo "💡 如果没有安装sshpass (Windows)，请手动执行以下步骤："
    echo "   1. 使用上面的命令连接服务器"
    echo "   2. 运行: bash < (curl -s https://raw.githubusercontent.com/...)"
    echo "   或参考下面的手动部署步骤"
    exit 1
}

# ========== 步骤2：创建远程目录 ==========
echo ""
echo "[2/6] 创建远程项目目录..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "mkdir -p ${REMOTE_DIR}"

# ========== 步骤3：同步项目文件 ==========
echo ""
echo "[3/6] 上传项目文件（这可能需要几分钟）..."
echo "排除大文件和结果目录..."

# 检查是否安装了rsync
if command -v rsync &> /dev/null; then
    echo "使用rsync同步..."
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
        --exclude '*.log' \
        ./ ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/
else
    echo "⚠️  rsync未安装，使用scp（较慢）..."
    echo "💡 建议安装rsync以加快上传速度"
    # 创建临时压缩包
    tar czf /tmp/vec_project.tar.gz \
        --exclude='__pycache__' \
        --exclude='*.pyc' \
        --exclude='.git' \
        --exclude='results' \
        --exclude='models' \
        --exclude='academic_figures' \
        --exclude='*.png' \
        --exclude='*.pdf' \
        --exclude='test_results' \
        .
    
    sshpass -p "${SERVER_PASSWORD}" scp -P ${SERVER_PORT} /tmp/vec_project.tar.gz ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/
    sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} "cd ${REMOTE_DIR} && tar xzf vec_project.tar.gz && rm vec_project.tar.gz"
    rm /tmp/vec_project.tar.gz
fi

# ========== 步骤4：配置服务器环境 ==========
echo ""
echo "[4/6] 配置服务器环境..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} << 'ENDSSH'
cd /root/VEC_mig_caching

echo "=========================================="
echo "环境检查"
echo "=========================================="

echo ""
echo "Python版本:"
python --version

echo ""
echo "CUDA版本:"
nvcc --version 2>/dev/null || echo "⚠️  CUDA未安装或不在PATH中"

echo ""
echo "GPU信息:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>/dev/null || echo "⚠️  无法检测GPU"

echo ""
echo "安装Python依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple || \
pip install -r requirements.txt

echo ""
echo "验证PyTorch和CUDA:"
python -c "
import torch
import sys
print(f'PyTorch版本: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU数量: {torch.cuda.device_count()}')
    print(f'GPU型号: {torch.cuda.get_device_name(0)}')
    print(f'显存大小: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
else:
    print('⚠️  警告: CUDA不可用，将使用CPU训练（非常慢）')
"

echo ""
echo "✅ 环境配置完成！"
ENDSSH

# ========== 步骤5：创建批量实验启动脚本 ==========
echo ""
echo "[5/6] 创建批量实验启动脚本..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} << 'ENDSSH'
cd /root/VEC_mig_caching

# ========== 批量实验启动脚本 ==========
cat > start_batch_experiments.sh << 'EOF'
#!/bin/bash
# 批量参数敏感性实验启动脚本

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="batch_experiments_${TIMESTAMP}.log"

echo "=========================================="
echo "VEC批量参数敏感性实验"
echo "开始时间: $(date)"
echo "=========================================="
echo ""
echo "实验配置:"
echo "  模式: full (500轮/配置)"
echo "  实验数: 8个参数对比"
echo "  预计时间: 2-5天"
echo "  日志文件: ${LOG_FILE}"
echo ""

# 启动训练（后台运行）
nohup python experiments/camtd3_strategy_suite/run_batch_experiments.py \
    --mode full \
    --all \
    --non-interactive \
    > ${LOG_FILE} 2>&1 &

PID=$!
echo "✅ 批量实验已在后台启动！"
echo ""
echo "进程信息:"
echo "  进程ID: ${PID}"
echo "  日志文件: ${LOG_FILE}"
echo ""
echo "监控命令:"
echo "  实时查看日志:  tail -f ${LOG_FILE}"
echo "  查看最后50行:  tail -50 ${LOG_FILE}"
echo "  查看进程状态:  ps aux | grep run_batch_experiments"
echo "  查看GPU使用:   nvidia-smi"
echo "  停止实验:      kill ${PID}"
echo ""
echo "或使用监控脚本:"
echo "  ./monitor_batch.sh"
echo ""

# 保存PID
echo ${PID} > batch_experiments.pid
EOF

chmod +x start_batch_experiments.sh

# ========== 监控脚本 ==========
cat > monitor_batch.sh << 'EOF'
#!/bin/bash
# 批量实验监控脚本

echo "=========================================="
echo "VEC批量实验监控"
echo "当前时间: $(date)"
echo "=========================================="

echo ""
echo "【进程状态】"
if [ -f batch_experiments.pid ]; then
    PID=$(cat batch_experiments.pid)
    if ps -p ${PID} > /dev/null 2>&1; then
        echo "✅ 实验正在运行 (PID: ${PID})"
        
        # 显示进程运行时间
        START_TIME=$(ps -p ${PID} -o lstart= 2>/dev/null)
        echo "   启动时间: ${START_TIME}"
        
        # 显示CPU和内存使用
        ps -p ${PID} -o pid,ppid,%cpu,%mem,etime,cmd --no-headers
    else
        echo "❌ 实验进程已停止 (PID: ${PID})"
    fi
else
    echo "⚠️  未找到进程ID文件"
    echo "   尝试查找运行中的实验..."
    ps aux | grep run_batch_experiments | grep -v grep
fi

echo ""
echo "【GPU使用情况】"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader 2>/dev/null || echo "⚠️  无法获取GPU信息"

echo ""
echo "【最新日志】(最后30行)"
LATEST_LOG=$(ls -t batch_experiments_*.log 2>/dev/null | head -1)
if [ -n "${LATEST_LOG}" ]; then
    echo "日志文件: ${LATEST_LOG}"
    echo "----------------------------------------"
    tail -30 ${LATEST_LOG}
else
    echo "⚠️  未找到日志文件"
fi

echo ""
echo "【结果目录】"
if [ -d "results/parameter_sensitivity" ]; then
    echo "results/parameter_sensitivity/"
    ls -lh results/parameter_sensitivity/ 2>/dev/null | tail -20
else
    echo "⚠️  结果目录尚未创建"
fi

echo ""
echo "=========================================="
echo "刷新: watch -n 10 ./monitor_batch.sh"
echo "停止: kill $(cat batch_experiments.pid 2>/dev/null)"
echo "=========================================="
EOF

chmod +x monitor_batch.sh

# ========== 快速测试脚本 ==========
cat > test_quick.sh << 'EOF'
#!/bin/bash
# 快速测试脚本 (10轮，用于验证功能)

echo "运行快速测试 (10轮/配置, 约2-3小时)..."
python experiments/camtd3_strategy_suite/run_batch_experiments.py \
    --mode quick \
    --all \
    --non-interactive
EOF

chmod +x test_quick.sh

echo "✅ 启动脚本创建完成！"
ENDSSH

# ========== 步骤6：启动实验 ==========
echo ""
echo "[6/6] 启动批量实验..."
sshpass -p "${SERVER_PASSWORD}" ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST} << 'ENDSSH'
cd /root/VEC_mig_caching
./start_batch_experiments.sh
ENDSSH

echo ""
echo "=========================================="
echo "✅ 部署并启动完成！"
echo "=========================================="
echo ""
echo "📊 实验信息:"
echo "   - 8个参数对比实验"
echo "   - 500轮/配置"
echo "   - 预计运行时间: 2-5天"
echo ""
echo "📝 监控方法:"
echo ""
echo "1️⃣  连接到服务器:"
echo "   ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}"
echo "   密码: ${SERVER_PASSWORD}"
echo ""
echo "2️⃣  查看实验状态:"
echo "   cd ${REMOTE_DIR}"
echo "   ./monitor_batch.sh          # 查看详细状态"
echo ""
echo "3️⃣  查看实时日志:"
echo "   tail -f batch_experiments_*.log"
echo ""
echo "4️⃣  查看GPU使用:"
echo "   watch -n 1 nvidia-smi"
echo ""
echo "5️⃣  下载结果 (实验完成后):"
echo "   scp -P ${SERVER_PORT} -r ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/results/parameter_sensitivity ./results_from_server"
echo ""
echo "6️⃣  如需停止实验:"
echo "   cd ${REMOTE_DIR}"
echo "   kill \$(cat batch_experiments.pid)"
echo ""
echo "=========================================="
echo ""
echo "💡 提示："
echo "   - 实验在后台运行，可以断开SSH连接"
echo "   - 定期登录查看进度和GPU使用情况"
echo "   - 建议保持服务器运行直到实验完成"
echo ""
echo "=========================================="

