#!/bin/bash
# ===================================================================
# VEC批量实验 - GPU加速部署脚本（交互式版本，无需sshpass）
# ===================================================================
# 服务器: connect.westc.gpuhub.com:21960
# 密码: B9iXNm5Ee0l4
# ===================================================================

set -e

# ========== 服务器配置 ==========
SERVER_HOST="connect.westc.gpuhub.com"
SERVER_PORT="21960"
SERVER_USER="root"
REMOTE_DIR="/root/VEC_mig_caching"

# ========== 颜色输出 ==========
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

echo "======================================================================="
echo "          VEC GPU训练自动化部署脚本（交互式版本）"
echo "======================================================================="
echo ""
echo "服务器信息:"
echo "  地址: $SERVER_HOST"
echo "  端口: $SERVER_PORT"
echo "  用户: $SERVER_USER"
echo "  密码: B9iXNm5Ee0l4"
echo ""
echo "======================================================================="
echo ""

# ========== 步骤1: 打包项目 ==========
info "步骤1: 打包项目文件..."

tar czf vec_project_gpu.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='results' \
    --exclude='models/improved_*' \
    --exclude='*.png' \
    --exclude='*.pdf' \
    --exclude='archives' \
    --exclude='*.tar.gz' \
    .

if [ -f "vec_project_gpu.tar.gz" ]; then
    FILE_SIZE=$(du -h vec_project_gpu.tar.gz | cut -f1)
    success "项目打包完成 (大小: $FILE_SIZE)"
else
    error "项目打包失败"
    exit 1
fi

# ========== 步骤2: 上传项目 ==========
info "步骤2: 上传项目到服务器..."
echo ""
warn "请在提示时输入密码: B9iXNm5Ee0l4"
echo ""

scp -P $SERVER_PORT -o StrictHostKeyChecking=no vec_project_gpu.tar.gz $SERVER_USER@$SERVER_HOST:/root/

if [ $? -eq 0 ]; then
    success "项目上传成功"
    rm vec_project_gpu.tar.gz
    info "本地压缩包已清理"
else
    error "项目上传失败"
    rm vec_project_gpu.tar.gz
    exit 1
fi

# ========== 步骤3: 创建远程执行脚本 ==========
info "步骤3: 创建远程部署脚本..."

cat > remote_setup.sh << 'REMOTESCRIPT'
#!/bin/bash

# 颜色定义
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

echo "======================================================================="
echo "                   服务器端部署开始"
echo "======================================================================="

# 解压项目
info "解压项目..."
cd /root
if [ -d "VEC_mig_caching" ]; then
    warn "备份现有目录..."
    mv VEC_mig_caching VEC_mig_caching_backup_$(date +%Y%m%d_%H%M%S)
fi

mkdir -p VEC_mig_caching
tar xzf vec_project_gpu.tar.gz -C VEC_mig_caching
cd VEC_mig_caching
success "项目解压完成"

# 检查环境
info "检查系统环境..."
echo "Python: $(python3 --version)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader)"

# 安装依赖
info "安装依赖..."
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple > /dev/null 2>&1

info "安装PyTorch (CUDA 11.8)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple > /dev/null 2>&1

info "安装其他依赖..."
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple > /dev/null 2>&1
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple > /dev/null 2>&1

success "依赖安装完成"

# 验证GPU
info "验证GPU..."
python3 << 'PYEOF'
import torch
import sys
print("=" * 70)
print("PyTorch版本:", torch.__version__)
print("CUDA可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU名称:", torch.cuda.get_device_name(0))
    print("GPU显存: {:.1f} GB".format(torch.cuda.get_device_properties(0).total_memory / 1024**3))
    print("=" * 70)
    x = torch.rand(1000, 1000).cuda()
    y = torch.matmul(x, x.T)
    print("✓ GPU测试通过")
    print("=" * 70)
else:
    print("✗ GPU不可用！")
    print("=" * 70)
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    error "GPU验证失败"
    exit 1
fi

# 创建训练脚本
info "创建GPU训练脚本..."
mkdir -p logs

cat > start_gpu_training.sh << 'TRAINEOF'
#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512

START_TIME=$(date +%s)
echo "训练开始: $(date)" | tee logs/training_start.log

python experiments/camtd3_strategy_suite/run_batch_experiments.py \
    --mode full \
    --all \
    --non-interactive \
    --silent 2>&1 | tee logs/batch_experiments_$(date +%Y%m%d_%H%M%S).log

END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "" | tee -a logs/training_start.log
echo "训练完成: $(date)" | tee -a logs/training_start.log
echo "总耗时: ${HOURS}小时${MINUTES}分钟" | tee -a logs/training_start.log

echo "5分钟后自动关机..."
shutdown -h +5 "VEC训练完成，系统将在5分钟后关机"
TRAINEOF

chmod +x start_gpu_training.sh

# 创建TensorBoard脚本
cat > start_tensorboard.sh << 'TBEOF'
#!/bin/bash
mkdir -p logs
nohup tensorboard --logdir=./results --port=6006 --bind_all > logs/tensorboard.log 2>&1 &
echo $! > tensorboard.pid
echo "TensorBoard已启动，访问: http://$(hostname -I | awk '{print $1}'):6006"
TBEOF

chmod +x start_tensorboard.sh

# 创建监控脚本
cat > monitor.sh << 'MONEOF'
#!/bin/bash
clear
echo "========================================"
echo "VEC GPU训练监控"
echo "========================================"
echo ""
echo "【GPU状态】"
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader | \
awk -F, '{printf "GPU使用:%s | 显存:%s/%s | 温度:%s\n", $1, $2, $3, $4}'
echo ""
echo "【训练进程】"
ps aux | grep "[r]un_batch_experiments" && echo "✓ 运行中" || echo "✗ 未运行"
echo ""
echo "【最新日志】"
tail -5 logs/batch_experiments_*.log 2>/dev/null || echo "暂无日志"
echo ""
echo "========================================"
MONEOF

chmod +x monitor.sh

success "所有脚本创建完成"

# 安装tmux
if ! command -v tmux &> /dev/null; then
    info "安装tmux..."
    apt-get update -qq && apt-get install -y tmux > /dev/null 2>&1
fi

# 启动TensorBoard
info "启动TensorBoard..."
./start_tensorboard.sh
sleep 2

# 在tmux中启动训练
info "在tmux会话中启动GPU训练..."
tmux new-session -d -s vec_training "./start_gpu_training.sh"

success "部署完成！"
echo ""
echo "======================================================================="
echo "                         部署成功！"
echo "======================================================================="
echo ""
echo "📊 监控命令:"
echo "   查看训练: tmux attach -t vec_training"
echo "   查看监控: ./monitor.sh"
echo "   查看GPU:  watch -n 1 nvidia-smi"
echo ""
echo "📁 日志位置:"
echo "   logs/batch_experiments_*.log"
echo ""
echo "🌐 TensorBoard:"
echo "   http://$(hostname -I | awk '{print $1}'):6006"
echo ""
echo "⏰ 训练完成后5分钟自动关机"
echo "   取消关机: shutdown -c"
echo ""
echo "======================================================================="

REMOTESCRIPT

success "远程脚本创建完成"

# ========== 步骤4: 上传并执行远程脚本 ==========
info "步骤4: 上传远程脚本..."
echo ""
warn "请输入密码: B9iXNm5Ee0l4"
echo ""

scp -P $SERVER_PORT -o StrictHostKeyChecking=no remote_setup.sh $SERVER_USER@$SERVER_HOST:/root/

if [ $? -eq 0 ]; then
    success "脚本上传成功"
    rm remote_setup.sh
else
    error "脚本上传失败"
    rm remote_setup.sh
    exit 1
fi

# ========== 步骤5: 执行远程部署 ==========
info "步骤5: 执行远程部署和训练启动..."
echo ""
warn "请输入密码: B9iXNm5Ee0l4"
echo ""

ssh -p $SERVER_PORT -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_HOST "cd /root && chmod +x remote_setup.sh && ./remote_setup.sh"

if [ $? -eq 0 ]; then
    echo ""
    echo "======================================================================="
    echo "                    🎉 部署成功！训练已启动"
    echo "======================================================================="
    echo ""
    echo "📱 连接服务器查看:"
    echo "   ssh -p $SERVER_PORT $SERVER_USER@$SERVER_HOST"
    echo "   密码: B9iXNm5Ee0l4"
    echo ""
    echo "🔍 进入训练会话:"
    echo "   tmux attach -t vec_training"
    echo ""
    echo "📊 查看监控:"
    echo "   ./monitor.sh"
    echo ""
    echo "📥 下载结果:"
    echo "   scp -P $SERVER_PORT -r $SERVER_USER@$SERVER_HOST:/root/VEC_mig_caching/results ./results_from_server"
    echo ""
    echo "⏰ 训练完成后5分钟自动关机"
    echo ""
    echo "======================================================================="
else
    error "部署失败"
    exit 1
fi

exit 0

