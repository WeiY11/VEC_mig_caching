#!/bin/bash
# ===================================================================
# VEC批量实验 - GPU加速自动化部署和训练脚本
# ===================================================================
# 服务器: connect.westc.gpuhub.com:21960
# 功能: 
# 1. 自动部署项目到服务器
# 2. 配置GPU环境
# 3. 启动批量实验（GPU加速）
# 4. 配置TensorBoard
# 5. 训练完成后自动关机
# ===================================================================

set -e  # 遇到错误立即退出

# ========== 服务器配置 ==========
SERVER_HOST="connect.westc.gpuhub.com"
SERVER_PORT="21960"
SERVER_USER="root"
SERVER_PASS="B9iXNm5Ee0l4"
REMOTE_DIR="/root/VEC_mig_caching"

# ========== 颜色输出 ==========
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

# ========== 步骤1: 测试服务器连接 ==========
info "步骤1: 测试服务器连接..."

sshpass -p "$SERVER_PASS" ssh -o StrictHostKeyChecking=no -p $SERVER_PORT $SERVER_USER@$SERVER_HOST "echo 'SSH连接成功'" 2>/dev/null
if [ $? -eq 0 ]; then
    success "服务器连接测试通过"
else
    error "无法连接到服务器，请检查网络和凭据"
    exit 1
fi

# ========== 步骤2: 打包项目文件 ==========
info "步骤2: 打包项目文件..."

tar czf vec_project_gpu.tar.gz \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='results' \
    --exclude='models/improved_*' \
    --exclude='*.png' \
    --exclude='*.pdf' \
    --exclude='archives' \
    --exclude='.vscode' \
    --exclude='node_modules' \
    .

if [ -f "vec_project_gpu.tar.gz" ]; then
    FILE_SIZE=$(du -h vec_project_gpu.tar.gz | cut -f1)
    success "项目打包完成 (大小: $FILE_SIZE)"
else
    error "项目打包失败"
    exit 1
fi

# ========== 步骤3: 上传项目到服务器 ==========
info "步骤3: 上传项目到服务器..."

sshpass -p "$SERVER_PASS" scp -P $SERVER_PORT -o StrictHostKeyChecking=no vec_project_gpu.tar.gz $SERVER_USER@$SERVER_HOST:/root/

if [ $? -eq 0 ]; then
    success "项目上传成功"
    rm vec_project_gpu.tar.gz
else
    error "项目上传失败"
    rm vec_project_gpu.tar.gz
    exit 1
fi

# ========== 步骤4: 在服务器上执行部署和训练 ==========
info "步骤4: 在服务器上部署并启动训练..."

sshpass -p "$SERVER_PASS" ssh -p $SERVER_PORT -o StrictHostKeyChecking=no $SERVER_USER@$SERVER_HOST << 'ENDSSH'

# 定义颜色
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info() { echo -e "${GREEN}[INFO]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }

echo "======================================================================="
echo "                   VEC GPU训练自动化部署脚本"
echo "======================================================================="

# ========== 4.1: 解压项目 ==========
info "解压项目文件..."
cd /root
rm -rf VEC_mig_caching_backup 2>/dev/null
if [ -d "VEC_mig_caching" ]; then
    warn "检测到已存在的项目目录，创建备份..."
    mv VEC_mig_caching VEC_mig_caching_backup
fi

tar xzf vec_project_gpu.tar.gz -C /root/VEC_mig_caching
cd /root/VEC_mig_caching
success "项目解压完成"

# ========== 4.2: 检查系统环境 ==========
info "检查系统环境..."

echo "Python版本:"
python3 --version

echo ""
echo "CUDA版本:"
if command -v nvcc &> /dev/null; then
    nvcc --version
else
    warn "未检测到CUDA编译器，但运行时可能可用"
fi

echo ""
echo "GPU信息:"
nvidia-smi || warn "nvidia-smi不可用"

# ========== 4.3: 安装/更新依赖 ==========
info "安装/更新Python依赖..."

pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装关键依赖
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装其他依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装tensorboard
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple

success "依赖安装完成"

# ========== 4.4: 验证GPU和PyTorch ==========
info "验证GPU和PyTorch配置..."

python3 << 'ENDPYTHON'
import torch
import sys

print("=" * 70)
print("PyTorch和CUDA验证")
print("=" * 70)

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA版本: {torch.version.cuda}")
    print(f"GPU数量: {torch.cuda.device_count()}")
    print(f"当前GPU: {torch.cuda.current_device()}")
    print(f"GPU名称: {torch.cuda.get_device_name(0)}")
    print(f"GPU显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # 测试GPU运算
    x = torch.rand(1000, 1000).cuda()
    y = torch.rand(1000, 1000).cuda()
    z = torch.matmul(x, y)
    print("GPU测试运算: 通过")
    print("=" * 70)
    sys.exit(0)
else:
    print("\n❌ 错误: CUDA不可用！")
    print("请检查:")
    print("1. NVIDIA驱动是否安装")
    print("2. PyTorch CUDA版本是否匹配")
    print("3. GPU是否被占用")
    print("=" * 70)
    sys.exit(1)
ENDPYTHON

if [ $? -ne 0 ]; then
    error "GPU验证失败，终止部署"
    exit 1
fi

success "GPU验证通过，可以使用CUDA加速"

# ========== 4.5: 创建训练启动脚本 ==========
info "创建GPU加速训练脚本..."

cat > /root/VEC_mig_caching/start_gpu_training.sh << 'ENDSCRIPT'
#!/bin/bash
# GPU加速批量实验启动脚本

# 设置CUDA环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb=512

# 创建日志目录
mkdir -p logs

# 记录开始时间
START_TIME=$(date +%s)
echo "实验开始时间: $(date)" | tee logs/training_start.log

# 运行批量实验
echo "======================================================================="
echo "启动GPU加速批量实验"
echo "======================================================================="

python experiments/camtd3_strategy_suite/run_batch_experiments.py \
    --mode full \
    --all \
    --non-interactive \
    --silent 2>&1 | tee logs/batch_experiments_$(date +%Y%m%d_%H%M%S).log

# 记录结束时间
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))

echo "" | tee -a logs/training_start.log
echo "=======================================================================" | tee -a logs/training_start.log
echo "实验完成！" | tee -a logs/training_start.log
echo "结束时间: $(date)" | tee -a logs/training_start.log
echo "总耗时: ${HOURS}小时${MINUTES}分钟" | tee -a logs/training_start.log
echo "=======================================================================" | tee -a logs/training_start.log

# 实验完成，准备关机
echo "实验已完成，5分钟后将自动关机..."
echo "如需取消关机，请运行: shutdown -c"

# 延迟5分钟后关机（给予下载结果的时间）
shutdown -h +5 "VEC实验完成，系统将在5分钟后关机"

exit 0
ENDSCRIPT

chmod +x /root/VEC_mig_caching/start_gpu_training.sh
success "训练脚本创建完成"

# ========== 4.6: 创建TensorBoard启动脚本 ==========
info "创建TensorBoard启动脚本..."

cat > /root/VEC_mig_caching/start_tensorboard.sh << 'ENDTB'
#!/bin/bash
# TensorBoard启动脚本

# 创建TensorBoard日志目录
mkdir -p /root/VEC_mig_caching/tensorboard_logs

# 启动TensorBoard
echo "启动TensorBoard..."
echo "访问地址: http://$(hostname -I | awk '{print $1}'):6006"

nohup tensorboard --logdir=/root/VEC_mig_caching/results \
                  --port=6006 \
                  --bind_all \
                  > /root/VEC_mig_caching/logs/tensorboard.log 2>&1 &

echo $! > /root/VEC_mig_caching/tensorboard.pid

echo "TensorBoard已启动，PID: $(cat /root/VEC_mig_caching/tensorboard.pid)"
echo "日志文件: /root/VEC_mig_caching/logs/tensorboard.log"
ENDTB

chmod +x /root/VEC_mig_caching/start_tensorboard.sh
success "TensorBoard脚本创建完成"

# ========== 4.7: 创建监控脚本 ==========
info "创建实验监控脚本..."

cat > /root/VEC_mig_caching/monitor_training.sh << 'ENDMONITOR'
#!/bin/bash
# 实验监控脚本

echo "======================================================================="
echo "                        VEC实验监控面板"
echo "======================================================================="

# GPU状态
echo ""
echo "【GPU状态】"
nvidia-smi --query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits | \
awk -F, '{printf "GPU%d: %s | 温度:%d°C | GPU使用:%d%% | 显存使用:%d%% | 显存:%dMB/%dMB\n", $1, $2, $3, $4, $5, $6, $7}'

# 进程状态
echo ""
echo "【训练进程】"
ps aux | grep "[r]un_batch_experiments.py" | head -1 && echo "✓ 批量实验运行中" || echo "✗ 批量实验未运行"
ps aux | grep "[t]ensorboard" | head -1 && echo "✓ TensorBoard运行中" || echo "✗ TensorBoard未运行"

# 系统资源
echo ""
echo "【系统资源】"
echo "CPU使用: $(top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}')"
echo "内存使用: $(free -h | awk '/^Mem:/ {print $3 "/" $2}')"
echo "磁盘使用: $(df -h / | awk 'NR==2 {print $3 "/" $2 " (" $5 ")"}')"

# 最新日志
echo ""
echo "【最新日志 (最后10行)】"
if [ -f logs/batch_experiments_*.log ]; then
    tail -10 $(ls -t logs/batch_experiments_*.log | head -1)
else
    echo "暂无日志文件"
fi

echo ""
echo "======================================================================="
echo "刷新时间: $(date)"
echo "======================================================================="
ENDMONITOR

chmod +x /root/VEC_mig_caching/monitor_training.sh
success "监控脚本创建完成"

# ========== 4.8: 安装tmux（如果需要） ==========
if ! command -v tmux &> /dev/null; then
    info "安装tmux..."
    apt-get update -qq && apt-get install -y tmux > /dev/null 2>&1
    success "tmux安装完成"
fi

# ========== 4.9: 启动TensorBoard ==========
info "启动TensorBoard..."
cd /root/VEC_mig_caching
./start_tensorboard.sh

sleep 2

if [ -f /root/VEC_mig_caching/tensorboard.pid ]; then
    success "TensorBoard已启动"
    echo "   访问地址: http://$(hostname -I | awk '{print $1}'):6006"
else
    warn "TensorBoard启动可能失败，请检查日志"
fi

# ========== 4.10: 在tmux会话中启动训练 ==========
info "在tmux会话中启动GPU加速训练..."

# 创建tmux会话并启动训练
tmux new-session -d -s vec_training "cd /root/VEC_mig_caching && ./start_gpu_training.sh"

if [ $? -eq 0 ]; then
    success "GPU训练已在tmux会话中启动"
    echo ""
    echo "======================================================================="
    echo "                           部署和启动完成！"
    echo "======================================================================="
    echo ""
    echo "📊 监控命令:"
    echo "   查看训练: tmux attach -t vec_training"
    echo "   查看监控: ./monitor_training.sh"
    echo "   查看GPU:  watch -n 1 nvidia-smi"
    echo ""
    echo "📁 日志位置:"
    echo "   训练日志: logs/batch_experiments_*.log"
    echo "   TensorBoard日志: logs/tensorboard.log"
    echo ""
    echo "🌐 TensorBoard:"
    echo "   地址: http://$(hostname -I | awk '{print $1}'):6006"
    echo ""
    echo "⏰ 关机设置:"
    echo "   训练完成后将在5分钟后自动关机"
    echo "   取消关机: shutdown -c"
    echo ""
    echo "📥 下载结果:"
    echo "   scp -P 21960 -r root@connect.westc.gpuhub.com:/root/VEC_mig_caching/results ./results_from_server"
    echo ""
    echo "======================================================================="
else
    error "训练启动失败"
    exit 1
fi

ENDSSH

if [ $? -eq 0 ]; then
    success "所有操作完成！"
    echo ""
    echo "======================================================================="
    echo "                      🎉 部署成功！训练已启动"
    echo "======================================================================="
    echo ""
    echo "📱 连接到服务器:"
    echo "   ssh -p 21960 root@connect.westc.gpuhub.com"
    echo ""
    echo "🔍 查看训练进度:"
    echo "   tmux attach -t vec_training"
    echo ""
    echo "📊 查看实时监控:"
    echo "   ./monitor_training.sh"
    echo ""
    echo "🌐 TensorBoard访问:"
    echo "   需要在服务器上查看IP地址"
    echo "   在服务器执行: hostname -I"
    echo ""
    echo "📥 实验完成后下载结果:"
    echo "   scp -P 21960 -r root@connect.westc.gpuhub.com:/root/VEC_mig_caching/results ./results_from_server"
    echo ""
    echo "⏰ 自动关机:"
    echo "   训练完成后服务器将在5分钟后自动关机"
    echo ""
    echo "======================================================================="
else
    error "部署失败，请检查错误信息"
    exit 1
fi

exit 0


