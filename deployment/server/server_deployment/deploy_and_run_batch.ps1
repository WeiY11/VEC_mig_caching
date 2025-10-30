# VEC项目 - 批量参数敏感性实验部署脚本 (PowerShell版本)
# 用途：部署到AutoDL服务器并运行完整的8个参数对比实验

# ========== 服务器配置 ==========
$SERVER_HOST = "region-9.autodl.pro"
$SERVER_PORT = "47042"
$SERVER_USER = "root"
$SERVER_PASSWORD = "dfUJkmli0mHk"
$REMOTE_DIR = "/root/VEC_mig_caching"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "VEC批量实验部署脚本 (Windows版)" -ForegroundColor Cyan
Write-Host "目标服务器: ${SERVER_USER}@${SERVER_HOST}:${SERVER_PORT}" -ForegroundColor Cyan
Write-Host "实验模式: full (500轮/配置, 预计2-5天)" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# ========== 检查依赖 ==========
Write-Host "检查依赖..." -ForegroundColor Yellow

# 检查是否安装了SSH客户端
$sshExists = Get-Command ssh -ErrorAction SilentlyContinue
if (-not $sshExists) {
    Write-Host "❌ 未找到SSH客户端！" -ForegroundColor Red
    Write-Host "请安装OpenSSH客户端或使用Git Bash" -ForegroundColor Yellow
    exit 1
}

# 检查是否安装了SCP
$scpExists = Get-Command scp -ErrorAction SilentlyContinue
if (-not $scpExists) {
    Write-Host "❌ 未找到SCP客户端！" -ForegroundColor Red
    exit 1
}

Write-Host "✅ SSH/SCP客户端已安装" -ForegroundColor Green

# ========== 创建临时密码文件 (不推荐，仅用于自动化) ==========
Write-Host ""
Write-Host "[1/5] 准备连接..." -ForegroundColor Yellow

# ========== 测试连接 ==========
Write-Host ""
Write-Host "💡 由于Windows限制，需要手动输入密码进行连接测试..." -ForegroundColor Yellow
Write-Host "密码: ${SERVER_PASSWORD}" -ForegroundColor Cyan
Write-Host ""

$testCmd = "ssh -p $SERVER_PORT -o StrictHostKeyChecking=no ${SERVER_USER}@${SERVER_HOST} 'echo 连接成功'"
Write-Host "执行: $testCmd" -ForegroundColor Gray

# 用户需要手动输入密码
ssh -p $SERVER_PORT -o StrictHostKeyChecking=no "${SERVER_USER}@${SERVER_HOST}" "echo '✅ 连接成功'"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "❌ 连接失败！请检查服务器信息" -ForegroundColor Red
    Write-Host ""
    Write-Host "💡 建议使用以下方式之一：" -ForegroundColor Yellow
    Write-Host "   1. 使用Git Bash运行 deploy_and_run_batch.sh" -ForegroundColor White
    Write-Host "   2. 手动部署 (见下方步骤)" -ForegroundColor White
    Write-Host "   3. 配置SSH密钥认证 (无需密码)" -ForegroundColor White
    Write-Host ""
    exit 1
}

Write-Host ""
Write-Host "=========================================="
Write-Host "手动部署步骤 (推荐)"
Write-Host "=========================================="
Write-Host ""
Write-Host "由于Windows PowerShell的限制，建议使用以下方法之一："
Write-Host ""
Write-Host "【方法1：使用Git Bash (推荐)】"
Write-Host "1. 打开Git Bash"
Write-Host "2. 进入项目目录: cd /d/VEC_mig_caching"
Write-Host "3. 运行部署脚本: bash deploy_and_run_batch.sh"
Write-Host ""
Write-Host "【方法2：使用WinSCP + PuTTY】"
Write-Host "1. 使用WinSCP上传整个项目文件夹到服务器"
Write-Host "   主机: ${SERVER_HOST}"
Write-Host "   端口: ${SERVER_PORT}"
Write-Host "   用户: ${SERVER_USER}"
Write-Host "   密码: ${SERVER_PASSWORD}"
Write-Host ""
Write-Host "2. 使用PuTTY连接服务器"
Write-Host ""
Write-Host "3. 在服务器上运行以下命令："
Write-Host "   cd ${REMOTE_DIR}"
Write-Host "   pip install -r requirements.txt"
Write-Host "   nohup python experiments/camtd3_strategy_suite/run_batch_experiments.py \"
Write-Host "       --mode full --all --non-interactive \"
Write-Host "       > batch_exp.log 2>&1 &"
Write-Host ""
Write-Host "【方法3：手动SSH命令】"
Write-Host ""
Write-Host "1️⃣  连接到服务器:"
Write-Host "   ssh -p ${SERVER_PORT} ${SERVER_USER}@${SERVER_HOST}" -ForegroundColor Green
Write-Host "   密码: ${SERVER_PASSWORD}" -ForegroundColor Cyan
Write-Host ""
Write-Host "2️⃣  上传项目文件 (在本地PowerShell中运行):"
Write-Host "   scp -P ${SERVER_PORT} -r . ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}" -ForegroundColor Green
Write-Host ""
Write-Host "3️⃣  在服务器上配置环境:"
Write-Host @"
cd ${REMOTE_DIR}
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}')"
"@ -ForegroundColor Green
Write-Host ""
Write-Host "4️⃣  启动批量实验 (后台运行):"
Write-Host @"
nohup python experiments/camtd3_strategy_suite/run_batch_experiments.py \
    --mode full \
    --all \
    --non-interactive \
    > batch_experiments.log 2>&1 &

echo `$! > batch_experiments.pid
"@ -ForegroundColor Green
Write-Host ""
Write-Host "5️⃣  监控实验进度:"
Write-Host "   tail -f batch_experiments.log" -ForegroundColor Green
Write-Host "   nvidia-smi" -ForegroundColor Green
Write-Host ""
Write-Host "6️⃣  下载结果 (实验完成后，在本地运行):"
Write-Host "   scp -P ${SERVER_PORT} -r ${SERVER_USER}@${SERVER_HOST}:${REMOTE_DIR}/results/parameter_sensitivity ./results_from_server" -ForegroundColor Green
Write-Host ""
Write-Host "=========================================="
Write-Host ""
Write-Host "💡 实验信息:" -ForegroundColor Yellow
Write-Host "   - 8个参数对比实验"
Write-Host "   - 每个配置500轮训练"
Write-Host "   - 预计运行时间: 2-5天"
Write-Host "   - 实验会在后台运行，可以断开SSH"
Write-Host ""
Write-Host "=========================================="

