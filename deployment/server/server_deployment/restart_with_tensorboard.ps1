# ================================================================
# 重新启动实验（带 TensorBoard 支持）- PowerShell 版本
# ================================================================
# 
# 【功能】
# 1. 停止旧的实验进程
# 2. 清理旧日志和 TensorBoard 数据（可选）
# 3. 启动 TensorBoard 服务
# 4. 启动新的批量实验
# 5. 提供监控命令
#
# 【使用方法】
# .\restart_with_tensorboard.ps1 [-Clean]
#
# 【参数】
# -Clean: 清理旧的日志和 TensorBoard 数据
#
# ================================================================

param(
    [switch]$Clean
)

# ========== 服务器配置 ==========
$SERVER_HOST = "region-9.autodl.pro"
$SERVER_PORT = "47042"
$SERVER_USER = "root"
$SERVER_PASSWORD = "dfUJkmli0mHk"
$REMOTE_DIR = "/root/VEC_mig_caching"

# ========== 颜色输出函数 ==========
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

# ========== 步骤1: 停止旧进程 ==========
Write-ColorOutput Yellow "[步骤 1/5] 停止旧的实验进程..."
sshpass -p "$SERVER_PASSWORD" ssh -p $SERVER_PORT "${SERVER_USER}@${SERVER_HOST}" "pkill -f run_batch_experiments; echo '旧进程已停止'"

if ($LASTEXITCODE -eq 0) {
    Write-ColorOutput Green "✓ 旧进程已停止"
} else {
    Write-ColorOutput Red "✗ 停止进程失败（可能没有运行中的进程）"
}

# ========== 步骤2: 清理旧数据（可选）==========
if ($Clean) {
    Write-ColorOutput Yellow "[步骤 2/5] 清理旧日志和 TensorBoard 数据..."
    sshpass -p "$SERVER_PASSWORD" ssh -p $SERVER_PORT "${SERVER_USER}@${SERVER_HOST}" "cd $REMOTE_DIR && rm -f batch_experiments.log && rm -rf runs/batch_experiments/* && echo '旧数据已清理'"
    
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✓ 旧数据已清理"
    } else {
        Write-ColorOutput Red "✗ 清理数据失败"
    }
} else {
    Write-ColorOutput Yellow "[步骤 2/5] 跳过清理旧数据（使用 -Clean 参数启用）"
}

# ========== 步骤3: 设置 TensorBoard ==========
Write-ColorOutput Yellow "[步骤 3/5] 设置 TensorBoard..."
sshpass -p "$SERVER_PASSWORD" ssh -p $SERVER_PORT "${SERVER_USER}@${SERVER_HOST}" "cd $REMOTE_DIR && bash server_deployment/setup_autodl_tensorboard.sh"

if ($LASTEXITCODE -eq 0) {
    Write-ColorOutput Green "✓ TensorBoard 设置完成"
} else {
    Write-ColorOutput Red "✗ TensorBoard 设置失败"
}

# ========== 步骤4: 启动 TensorBoard ==========
Write-ColorOutput Yellow "[步骤 4/5] 启动 TensorBoard 服务..."
sshpass -p "$SERVER_PASSWORD" ssh -p $SERVER_PORT "${SERVER_USER}@${SERVER_HOST}" "cd $REMOTE_DIR && bash server_deployment/start_tensorboard.sh"

if ($LASTEXITCODE -eq 0) {
    Write-ColorOutput Green "✓ TensorBoard 服务已启动（端口 6006）"
} else {
    Write-ColorOutput Red "✗ TensorBoard 启动失败"
}

# ========== 步骤5: 启动新实验 ==========
Write-ColorOutput Yellow "[步骤 5/5] 启动新的批量实验..."
sshpass -p "$SERVER_PASSWORD" ssh -p $SERVER_PORT "${SERVER_USER}@${SERVER_HOST}" "cd $REMOTE_DIR && bash server_deployment/remote_start.sh"

if ($LASTEXITCODE -eq 0) {
    Write-ColorOutput Green "✓ 批量实验已启动"
} else {
    Write-ColorOutput Red "✗ 实验启动失败"
    exit 1
}

# ========== 显示监控信息 ==========
Write-Host ""
Write-ColorOutput Green "========================================"
Write-ColorOutput Green "   实验重启成功！"
Write-ColorOutput Green "========================================"
Write-Host ""
Write-ColorOutput Yellow "📊 TensorBoard 访问方式："
Write-Host "1. AutoDL 控制台 → 自定义服务 → TensorBoard (端口 6006)"
Write-Host "2. SSH 隧道: ssh -p $SERVER_PORT -L 6006:localhost:6006 ${SERVER_USER}@${SERVER_HOST}"
Write-Host "   然后访问: http://localhost:6006"
Write-Host ""
Write-ColorOutput Yellow "🔍 监控实验进度："
Write-Host "查看日志:"
Write-Host "  sshpass -p '$SERVER_PASSWORD' ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} 'tail -f $REMOTE_DIR/batch_experiments.log'"
Write-Host ""
Write-Host "运行监控脚本:"
Write-Host "  sshpass -p '$SERVER_PASSWORD' ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} 'cd $REMOTE_DIR && bash server_deployment/remote_monitor.sh'"
Write-Host ""
Write-Host "检查进程状态:"
Write-Host "  sshpass -p '$SERVER_PASSWORD' ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} 'ps aux | grep run_batch'"
Write-Host ""
Write-Host "检查 GPU 使用:"
Write-Host "  sshpass -p '$SERVER_PASSWORD' ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} 'nvidia-smi'"
Write-Host ""

