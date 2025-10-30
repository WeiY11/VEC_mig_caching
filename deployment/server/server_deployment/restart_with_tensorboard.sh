#!/bin/bash
# ================================================================
# 重新启动实验（带 TensorBoard 支持）
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
# bash restart_with_tensorboard.sh [--clean]
#
# 【参数】
# --clean: 清理旧的日志和 TensorBoard 数据
#
# ================================================================

# ========== 服务器配置 ==========
SERVER_HOST="region-9.autodl.pro"
SERVER_PORT="47042"
SERVER_USER="root"
SERVER_PASSWORD="dfUJkmli0mHk"
REMOTE_DIR="/root/VEC_mig_caching"

# ========== 颜色输出 ==========
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# ========== 检查参数 ==========
CLEAN_OLD_DATA=false
if [[ "$1" == "--clean" ]]; then
    CLEAN_OLD_DATA=true
fi

# ========== 步骤1: 停止旧进程 ==========
echo -e "${YELLOW}[步骤 1/5] 停止旧的实验进程...${NC}"
sshpass -p "$SERVER_PASSWORD" ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} << 'ENDSSH'
    pkill -f run_batch_experiments
    echo "旧进程已停止"
ENDSSH

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 旧进程已停止${NC}"
else
    echo -e "${RED}✗ 停止进程失败（可能没有运行中的进程）${NC}"
fi

# ========== 步骤2: 清理旧数据（可选）==========
if [ "$CLEAN_OLD_DATA" = true ]; then
    echo -e "${YELLOW}[步骤 2/5] 清理旧日志和 TensorBoard 数据...${NC}"
    sshpass -p "$SERVER_PASSWORD" ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} << ENDSSH
        cd $REMOTE_DIR
        rm -f batch_experiments.log
        rm -rf runs/batch_experiments/*
        echo "旧数据已清理"
ENDSSH
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ 旧数据已清理${NC}"
    else
        echo -e "${RED}✗ 清理数据失败${NC}"
    fi
else
    echo -e "${YELLOW}[步骤 2/5] 跳过清理旧数据（使用 --clean 参数启用）${NC}"
fi

# ========== 步骤3: 设置 TensorBoard ==========
echo -e "${YELLOW}[步骤 3/5] 设置 TensorBoard...${NC}"
sshpass -p "$SERVER_PASSWORD" ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} << ENDSSH
    cd $REMOTE_DIR
    bash server_deployment/setup_autodl_tensorboard.sh
ENDSSH

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ TensorBoard 设置完成${NC}"
else
    echo -e "${RED}✗ TensorBoard 设置失败${NC}"
fi

# ========== 步骤4: 启动 TensorBoard ==========
echo -e "${YELLOW}[步骤 4/5] 启动 TensorBoard 服务...${NC}"
sshpass -p "$SERVER_PASSWORD" ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} << ENDSSH
    cd $REMOTE_DIR
    bash server_deployment/start_tensorboard.sh
ENDSSH

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ TensorBoard 服务已启动（端口 6006）${NC}"
else
    echo -e "${RED}✗ TensorBoard 启动失败${NC}"
fi

# ========== 步骤5: 启动新实验 ==========
echo -e "${YELLOW}[步骤 5/5] 启动新的批量实验...${NC}"
sshpass -p "$SERVER_PASSWORD" ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} << ENDSSH
    cd $REMOTE_DIR
    bash server_deployment/remote_start.sh
ENDSSH

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ 批量实验已启动${NC}"
else
    echo -e "${RED}✗ 实验启动失败${NC}"
    exit 1
fi

# ========== 显示监控信息 ==========
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   实验重启成功！${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo -e "${YELLOW}📊 TensorBoard 访问方式：${NC}"
echo "1. AutoDL 控制台 → 自定义服务 → TensorBoard (端口 6006)"
echo "2. SSH 隧道: ssh -p $SERVER_PORT -L 6006:localhost:6006 ${SERVER_USER}@${SERVER_HOST}"
echo "   然后访问: http://localhost:6006"
echo ""
echo -e "${YELLOW}🔍 监控实验进度：${NC}"
echo "查看日志:"
echo "  sshpass -p '$SERVER_PASSWORD' ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} 'tail -f $REMOTE_DIR/batch_experiments.log'"
echo ""
echo "运行监控脚本:"
echo "  sshpass -p '$SERVER_PASSWORD' ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} 'cd $REMOTE_DIR && bash server_deployment/remote_monitor.sh'"
echo ""
echo "检查进程状态:"
echo "  sshpass -p '$SERVER_PASSWORD' ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} 'ps aux | grep run_batch'"
echo ""
echo "检查 GPU 使用:"
echo "  sshpass -p '$SERVER_PASSWORD' ssh -p $SERVER_PORT ${SERVER_USER}@${SERVER_HOST} 'nvidia-smi'"
echo ""

