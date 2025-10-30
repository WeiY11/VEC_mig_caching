#!/bin/bash
# 配置AutoDL控制台的TensorBoard

cd /root/VEC_mig_caching

echo "=========================================="
echo "配置AutoDL TensorBoard"
echo "=========================================="

# 创建训练日志目录（如果不存在）
mkdir -p runs/batch_experiments

# 清理AutoDL默认目录的旧内容
rm -rf /root/tf-logs/*

# 创建软链接到AutoDL的TensorBoard目录
echo "创建软链接到 /root/tf-logs/ ..."
ln -sf /root/VEC_mig_caching/runs/batch_experiments /root/tf-logs/vec_batch_experiments

echo ""
echo "✅ 配置完成！"
echo ""
echo "【在AutoDL控制台查看TensorBoard】"
echo "1. 打开AutoDL控制台 (https://www.autodl.com/console)"
echo "2. 找到你的实例"
echo "3. 点击 'TensorBoard' 按钮"
echo "4. 会自动打开TensorBoard界面"
echo ""
echo "或者使用自定义访问："
echo "  在AutoDL控制台找到实例的JupyterLab链接"
echo "  将端口改为 6007 或 6006"
echo ""
echo "软链接已创建："
ls -la /root/tf-logs/
echo ""

