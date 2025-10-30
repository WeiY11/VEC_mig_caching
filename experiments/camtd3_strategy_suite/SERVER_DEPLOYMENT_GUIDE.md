# VEC批量实验服务器部署指南

## 📋 服务器信息

- **主机**: region-9.autodl.pro
- **端口**: 47042
- **用户**: root
- **密码**: dfUJkmli0mHk
- **远程目录**: /root/VEC_mig_caching

---

## 🚀 快速开始

### Windows用户 (Git Bash - 推荐)

```bash
# 在项目根目录下运行
bash deploy_and_run_batch.sh
```

这个脚本会自动：
1. ✅ 测试服务器连接
2. ✅ 上传项目文件
3. ✅ 配置Python环境
4. ✅ 启动批量实验（后台运行）

### Linux/Mac用户

```bash
# 在项目根目录下运行
bash deploy_and_run_batch.sh
```

---

## 📝 手动部署步骤

如果自动脚本失败，可以手动部署：

### 步骤1：连接到服务器

```bash
ssh -p 47042 root@region-9.autodl.pro
# 密码: dfUJkmli0mHk
```

### 步骤2：上传项目文件

**方法A: 使用SCP (本地Windows PowerShell/Git Bash)**

```bash
# 压缩项目（排除结果文件）
tar czf vec_project.tar.gz \
    --exclude='__pycache__' \
    --exclude='.git' \
    --exclude='results' \
    --exclude='*.png' \
    --exclude='*.pdf' \
    .

# 上传到服务器
scp -P 47042 vec_project.tar.gz root@region-9.autodl.pro:/root/

# 清理本地压缩包
rm vec_project.tar.gz
```

**方法B: 使用Git (在服务器上)**

```bash
# 在服务器上
cd /root
git clone <你的仓库地址> VEC_mig_caching
cd VEC_mig_caching
```

### 步骤3：配置环境

在服务器上运行：

```bash
cd /root/VEC_mig_caching

# 检查Python和CUDA
python --version
nvcc --version
nvidia-smi

# 安装依赖
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证PyTorch和CUDA
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA可用: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
"
```

### 步骤4：启动批量实验

```bash
cd /root/VEC_mig_caching

# 创建启动脚本
cat > start_batch.sh << 'EOF'
#!/bin/bash
nohup python experiments/camtd3_strategy_suite/run_batch_experiments.py \
    --mode full \
    --all \
    --non-interactive \
    > batch_experiments_$(date +%Y%m%d_%H%M%S).log 2>&1 &

echo $! > batch_experiments.pid
echo "✅ 实验已启动，PID: $(cat batch_experiments.pid)"
echo "查看日志: tail -f batch_experiments_*.log"
EOF

chmod +x start_batch.sh

# 启动实验
./start_batch.sh
```

---

## 📊 监控实验进度

### 方法1：使用监控脚本（推荐）

```bash
# 在服务器上
cd /root/VEC_mig_caching
./monitor_batch.sh
```

### 方法2：查看日志

```bash
# 实时查看日志
tail -f batch_experiments_*.log

# 查看最后100行
tail -100 batch_experiments_*.log

# 查找特定信息
grep "Episode" batch_experiments_*.log | tail -20
grep "实验" batch_experiments_*.log | tail -20
```

### 方法3：检查进程

```bash
# 查看实验进程
ps aux | grep run_batch_experiments

# 查看进程PID
cat batch_experiments.pid

# 查看进程详细信息
ps -p $(cat batch_experiments.pid) -f
```

### 方法4：查看GPU使用

```bash
# 实时GPU监控
watch -n 1 nvidia-smi

# 或简单查看
nvidia-smi
```

### 方法5：查看结果目录

```bash
# 查看生成的结果
ls -lh results/parameter_sensitivity/

# 查看最新结果
ls -lt results/parameter_sensitivity/ | head -10
```

---

## 🎯 实验配置

### Full模式（当前运行）

- **模式**: full
- **轮数**: 500轮/配置
- **实验数**: 8个参数对比
- **预计时间**: 2-5天
- **总配置数**: 47个配置
- **总训练轮次**: 23,500轮

### 8个参数对比实验

1. ✅ 数据大小对比 (5个配置)
2. ✅ 车辆数量对比 (5个配置)
3. ✅ 本地资源对卸载影响 (5个配置)
4. ✅ 本地资源对成本影响 (7个配置)
5. ✅ 带宽对成本影响 (7个配置)
6. ✅ 边缘节点配置对比 (6个配置)
7. ✅ 任务到达率对比 (6个配置)
8. ✅ 移动速度对比 (6个配置)

---

## 📥 下载结果

### 实验完成后

在**本地计算机**上运行：

```bash
# 下载所有结果
scp -P 47042 -r root@region-9.autodl.pro:/root/VEC_mig_caching/results/parameter_sensitivity ./results_from_server

# 或只下载特定实验
scp -P 47042 -r root@region-9.autodl.pro:/root/VEC_mig_caching/results/parameter_sensitivity/batch_full_* ./results_from_server

# 下载日志
scp -P 47042 root@region-9.autodl.pro:/root/VEC_mig_caching/batch_experiments_*.log ./logs_from_server/
```

### 结果文件结构

```
results/parameter_sensitivity/
└── batch_full_20241030_*/
    ├── summary.json              # 汇总数据
    ├── data_size_comparison.png  # 数据大小对比图
    ├── vehicle_count_comparison.png
    ├── bandwidth_cost_comparison.png
    ├── edge_node_comparison.png
    ├── edge_node_heatmap.png
    ├── task_arrival_comparison.png
    ├── mobility_speed_comparison.png
    └── ...                        # 其他图表和数据
```

---

## 🛑 停止实验

### 优雅停止

```bash
# 在服务器上
cd /root/VEC_mig_caching
kill $(cat batch_experiments.pid)
```

### 强制停止

```bash
# 查找进程
ps aux | grep run_batch_experiments

# 强制杀死
kill -9 <PID>
```

---

## 🔧 故障排除

### 问题1：连接超时

```bash
# 检查网络连接
ping region-9.autodl.pro

# 检查SSH服务
telnet region-9.autodl.pro 47042
```

### 问题2：CUDA不可用

```bash
# 检查NVIDIA驱动
nvidia-smi

# 检查CUDA版本
nvcc --version

# 重新安装PyTorch (CUDA 11.8示例)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 问题3：内存不足

```bash
# 查看内存使用
free -h

# 查看磁盘空间
df -h

# 清理缓存
rm -rf __pycache__
rm -rf .cache
```

### 问题4：进程意外终止

```bash
# 查看日志末尾
tail -50 batch_experiments_*.log

# 查看系统日志
dmesg | tail -50

# 重新启动实验
./start_batch.sh
```

### 问题5：依赖安装失败

```bash
# 使用国内镜像
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 或清华镜像
pip install -r requirements.txt -i https://pypi.mirrors.ustc.edu.cn/simple/

# 手动安装关键依赖
pip install torch numpy matplotlib pandas -i https://pypi.tuna.tsinghua.edu.cn/simple
```

---

## 📱 远程监控（可选）

### 使用Tmux会话

```bash
# 安装tmux
apt install tmux -y

# 创建会话
tmux new -s vec_training

# 在tmux中运行实验
./start_batch.sh

# 断开会话（实验继续运行）
# 按 Ctrl+B，然后按 D

# 重新连接
tmux attach -t vec_training
```

### 使用Screen会话

```bash
# 安装screen
apt install screen -y

# 创建会话
screen -S vec_training

# 在screen中运行实验
./start_batch.sh

# 断开会话
# 按 Ctrl+A，然后按 D

# 重新连接
screen -r vec_training
```

---

## 📧 实验完成通知（可选）

可以修改脚本添加邮件或webhook通知：

```bash
# 在start_batch.sh末尾添加
cat >> start_batch.sh << 'EOF'

# 等待实验完成
wait $PID

# 发送通知（示例）
curl -X POST https://your-webhook-url.com/notify \
    -d "message=VEC实验完成！"
EOF
```

---

## ⏱️ 预计时间表

| 阶段 | 时间 | 说明 |
|------|------|------|
| 部署和环境配置 | 10-30分钟 | 上传文件+安装依赖 |
| 实验1-2 | 12-16小时 | 数据大小、车辆数量 |
| 实验3-5 | 18-24小时 | 本地资源、带宽影响 |
| 实验6-8 | 16-20小时 | 边缘节点、任务到达、移动速度 |
| **总计** | **2-3天** | 取决于GPU性能 |

---

## 💡 最佳实践

1. ✅ **使用tmux/screen**：避免SSH断开导致实验中断
2. ✅ **定期检查**：每天登录查看进度和GPU状态
3. ✅ **保存日志**：定期下载日志备份
4. ✅ **监控GPU温度**：确保不过热（<85°C）
5. ✅ **预留磁盘空间**：至少20GB可用空间
6. ✅ **网络稳定性**：使用稳定的服务器实例

---

## 📞 联系支持

如有问题，可以：
1. 查看日志文件获取详细错误信息
2. 检查GPU和系统资源
3. 参考本文档的故障排除章节
4. 查看项目README和文档

---

**祝实验顺利！🎉**

