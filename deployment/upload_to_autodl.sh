#!/bin/bash
# 快速上传项目到AutoDL服务器

SERVER="root@region-41.seetacloud.com"
PORT="38597"
REMOTE_DIR="/root/VEC_mig_caching"

echo "=========================================="
echo "上传项目到AutoDL服务器"
echo "=========================================="
echo ""
echo "目标: ${SERVER}:${REMOTE_DIR}"
echo "密码: dXI7ldI+vPec"
echo ""

# 使用rsync上传
rsync -avz --progress \
    -e "ssh -p ${PORT}" \
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
    --exclude 'deployment/' \
    ./ ${SERVER}:${REMOTE_DIR}/

echo ""
echo "=========================================="
echo "✅ 上传完成！"
echo "=========================================="
echo ""
echo "下一步："
echo "1. 连接服务器: ssh -p ${PORT} ${SERVER}"
echo "2. 进入目录: cd ${REMOTE_DIR}"
echo "3. 验证文件: ls -la"
echo ""
