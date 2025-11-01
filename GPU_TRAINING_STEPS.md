# GPU训练部署步骤（更新版）

## 服务器信息
```
地址: connect.westc.gpuhub.com
端口: 21960
用户: root
密码: B9iXNm5Ee0l4
```

---

## 📋 PowerShell手动操作步骤

### 第1步：上传文件

在PowerShell中运行：

```powershell
cd D:\VEC_mig_caching
scp -P 21960 archives\vec_project.tar.gz root@connect.westc.gpuhub.com:/root/
```

**输入密码**：`B9iXNm5Ee0l4`

---

### 第2步：连接服务器

```powershell
ssh -p 21960 root@connect.westc.gpuhub.com
```

**输入密码**：`B9iXNm5Ee0l4`

---

### 第3步：在服务器上部署

**连接成功后，复制粘贴以下整段命令**：

```bash
cd /root && \
tar xzf vec_project.tar.gz -C /root/VEC_mig_caching && \
cd /root/VEC_mig_caching && \
pip install --upgrade pip -i https://pypi.tuna.tsinghua.edu.cn/simple && \
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 && \
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple && \
pip install tensorboard -i https://pypi.tuna.tsinghua.edu.cn/simple && \
echo "环境配置完成！"
```

---

### 第4步：验证GPU并启动训练

**复制粘贴以下整段命令**：

```bash
python3 -c "import torch; print('CUDA:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')" && \
apt-get update && apt-get install -y tmux && \
cd /root/VEC_mig_caching && \
mkdir -p logs && \
tmux new -s vec_training "export CUDA_VISIBLE_DEVICES=0 && python experiments/camtd3_strategy_suite/run_batch_experiments.py --mode full --all --non-interactive --silent 2>&1 | tee logs/training_\$(date +%Y%m%d_%H%M%S).log && shutdown -h +5"
```

**训练启动后**：
1. 按 `Ctrl+B`，然后按 `D` 断开tmux
2. 可以安全关闭SSH连接，训练继续运行

---

## 🔍 监控命令（可选）

随时可以重新连接查看进度：

```bash
# 重新SSH连接
ssh -p 21960 root@connect.westc.gpuhub.com

# 重新进入训练会话
tmux attach -t vec_training

# 或查看GPU状态
watch -n 1 nvidia-smi

# 查看训练日志
tail -f /root/VEC_mig_caching/logs/training_*.log
```

---

## 📥 下载结果

训练完成后，在本地PowerShell运行：

```powershell
cd D:\VEC_mig_caching
scp -P 21960 -r root@connect.westc.gpuhub.com:/root/VEC_mig_caching/results ./results_from_server
scp -P 21960 -r root@connect.westc.gpuhub.com:/root/VEC_mig_caching/logs ./logs_from_server
```

---

## 🆘 常用命令

| 操作 | 命令 |
|------|------|
| 取消自动关机 | `shutdown -c` |
| 停止训练 | `pkill -f run_batch` |
| 查看进程 | `ps aux \| grep run_batch` |
| 退出tmux | `Ctrl+B` 然后按 `D` |
| 强制退出tmux | `Ctrl+B` 然后输入 `:kill-session` |

---

## ⚠️ 重要说明

- ✅ **GPU加速**：已配置 `CUDA_VISIBLE_DEVICES=0`
- ✅ **自动关机**：训练完成5分钟后自动关机（节省费用）
- ✅ **TensorBoard**：训练日志自动保存，可后期可视化
- ⏱️ **预计时间**：2-3天（具体取决于GPU型号）

---

**现在可以从第1步开始了！** 🚀

