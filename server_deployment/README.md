# 🚀 服务器部署与监控指南

## 📦 文件清单

### 部署脚本
- `deploy_and_run_batch.sh` - Linux/Mac/Git Bash 部署脚本
- `deploy_and_run_batch.ps1` - Windows PowerShell 部署脚本
- `check_env.py` - 环境检查脚本

### 远程控制脚本
- `remote_start.sh` - 服务器上的启动脚本（自动生成）
- `remote_monitor.sh` - 服务器上的监控脚本（自动生成）
- `start_remote_batch.py` - 本地远程启动辅助脚本

### TensorBoard 相关
- `setup_tensorboard.py` - TensorBoard 目录初始化
- `start_tensorboard.sh` - 启动 TensorBoard 服务
- `setup_autodl_tensorboard.sh` - AutoDL 平台 TensorBoard 配置
- `monitor_to_tensorboard.py` - 日志到 TensorBoard 转换器
- `start_tb_monitor.sh` - 启动日志监控

### 文档
- `DEPLOYMENT_SUCCESS.md` - 部署成功说明
- `QUICK_MONITOR_GUIDE.txt` - 快速监控指南
- `TENSORBOARD_GUIDE.md` - TensorBoard 使用指南

---

## 🎯 快速开始

### 方式1：完整部署（推荐）

**Windows PowerShell**：
```powershell
cd server_deployment
.\deploy_and_run_batch.ps1
```

**Linux/Mac/Git Bash**：
```bash
cd server_deployment
bash deploy_and_run_batch.sh
```

### 方式2：仅重启实验（服务器已部署）

```bash
# 先停止旧进程
ssh -p 47042 root@region-9.autodl.pro "pkill -f run_batch_experiments"

# 重新启动
ssh -p 47042 root@region-9.autodl.pro "cd /root/VEC_mig_caching && nohup /root/miniconda3/bin/python experiments/camtd3_strategy_suite/run_batch_experiments.py --mode full --all --non-interactive > batch_experiments.log 2>&1 &"
```

---

## 📊 TensorBoard 启动步骤

### AutoDL 平台自动监控

1. **在服务器上设置 TensorBoard**：
```bash
ssh -p 47042 root@region-9.autodl.pro << 'ENDSSH'
cd /root/VEC_mig_caching
bash server_deployment/setup_autodl_tensorboard.sh
bash server_deployment/start_tensorboard.sh
ENDSSH
```

2. **在 AutoDL 控制台**：
   - 点击"自定义服务" → "TensorBoard"
   - 端口：6006
   - 会自动打开 TensorBoard Web 界面

### 本地 SSH 隧道访问

如果 AutoDL 控制台访问不便：

```bash
# 建立 SSH 隧道
ssh -p 47042 -L 6006:localhost:6006 root@region-9.autodl.pro

# 然后在浏览器访问：http://localhost:6006
```

---

## 🔍 监控实验进度

### 方法1：查看实时日志
```bash
ssh -p 47042 root@region-9.autodl.pro "tail -f /root/VEC_mig_caching/batch_experiments.log"
```

### 方法2：运行监控脚本
```bash
ssh -p 47042 root@region-9.autodl.pro "cd /root/VEC_mig_caching && bash server_deployment/remote_monitor.sh"
```

### 方法3：TensorBoard 可视化
访问 http://localhost:6006（如果配置了 SSH 隧道）

---

## 🛠️ 常见问题

### Q1: TensorBoard 没有数据？
**原因**：训练脚本可能未配置 TensorBoard writer

**解决方案**：
1. 确保 `train_single_agent.py` 使用了 `torch.utils.tensorboard.SummaryWriter`
2. 检查 `runs/batch_experiments/` 目录是否有事件文件
3. 重新启动实验

### Q2: 部署后实验未启动？
```bash
# 检查进程
ssh -p 47042 root@region-9.autodl.pro "ps aux | grep run_batch"

# 如果没有，手动启动
ssh -p 47042 root@region-9.autodl.pro "cd /root/VEC_mig_caching && bash server_deployment/remote_start.sh"
```

### Q3: GPU 未使用？
```bash
# 检查 CUDA 可用性
ssh -p 47042 root@region-9.autodl.pro "cd /root/VEC_mig_caching && /root/miniconda3/bin/python -c 'import torch; print(torch.cuda.is_available())'"

# 检查 GPU 使用情况
ssh -p 47042 root@region-9.autodl.pro "nvidia-smi"
```

---

## 📈 实验配置

### 当前批量实验设置
- **模式**: full (完整实验，200轮/实验)
- **实验数量**: 8个参数敏感性分析
- **预计时间**: 约 6-10 小时
- **日志位置**: `/root/VEC_mig_caching/batch_experiments.log`

### 快速测试模式
如需快速验证（每实验仅30轮）：
```bash
ssh -p 47042 root@region-9.autodl.pro "cd /root/VEC_mig_caching && /root/miniconda3/bin/python experiments/camtd3_strategy_suite/run_batch_experiments.py --mode quick --all --non-interactive"
```

---

## 🔄 重新运行实验（带 TensorBoard）

### 完整流程

1. **停止旧进程**：
```bash
sshpass -p 'dfUJkmli0mHk' ssh -p 47042 root@region-9.autodl.pro "pkill -f run_batch_experiments"
```

2. **清理旧日志**（可选）：
```bash
sshpass -p 'dfUJkmli0mHk' ssh -p 47042 root@region-9.autodl.pro "cd /root/VEC_mig_caching && rm -f batch_experiments.log && rm -rf runs/batch_experiments/*"
```

3. **启动 TensorBoard**：
```bash
sshpass -p 'dfUJkmli0mHk' ssh -p 47042 root@region-9.autodl.pro "cd /root/VEC_mig_caching && bash server_deployment/setup_autodl_tensorboard.sh && bash server_deployment/start_tensorboard.sh"
```

4. **启动新实验**：
```bash
sshpass -p 'dfUJkmli0mHk' ssh -p 47042 root@region-9.autodl.pro "cd /root/VEC_mig_caching && bash server_deployment/remote_start.sh"
```

5. **监控进度**：
- TensorBoard: AutoDL 控制台 → 自定义服务 → TensorBoard (端口6006)
- 日志: `tail -f batch_experiments.log`

---

## 📞 服务器信息

```
主机: region-9.autodl.pro
端口: 47042
用户: root
密码: dfUJkmli0mHk
项目目录: /root/VEC_mig_caching
```

---

## ✅ 检查清单

部署成功后应确认：
- [ ] SSH 连接正常
- [ ] Python 环境正确（3.8+）
- [ ] CUDA 可用（`nvidia-smi` 有输出）
- [ ] PyTorch CUDA 可用（`torch.cuda.is_available()` 返回 True）
- [ ] 实验进程运行中（`ps aux | grep run_batch`）
- [ ] 日志文件更新（`tail batch_experiments.log`）
- [ ] TensorBoard 启动（端口6006监听）
- [ ] GPU 使用率 > 0%（`nvidia-smi`）

---

**最后更新**: 2025-10-30  
**版本**: v1.1 (支持 TensorBoard)

