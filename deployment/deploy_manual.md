# 🚀 VEC项目远程服务器部署指南

## 📋 服务器信息
- **主机**: region-9.autodl.pro
- **端口**: 19287
- **用户**: root
- **密码**: dfUJkmli0mHk
- **项目目录**: /root/VEC_mig_caching

---

## 🎯 快速部署（推荐）

### 方案A：自动化脚本（Linux/Mac）

1. **安装sshpass工具**（如果未安装）：
```bash
# Ubuntu/Debian
sudo apt install sshpass

# MacOS
brew install hudochenkov/sshpass/sshpass
```

2. **运行部署脚本**：
```bash
chmod +x deploy_to_server.sh
./deploy_to_server.sh
```

3. **完成！** 脚本会自动完成：
   - ✅ 上传项目文件
   - ✅ 配置环境
   - ✅ 安装依赖
   - ✅ 创建训练脚本

---

### 方案B：手动部署（Windows或不想安装sshpass）

#### 步骤1: 连接服务器
```bash
ssh -p 19287 root@region-9.autodl.pro
# 输入密码: dfUJkmli0mHk
```

#### 步骤2: 创建项目目录
```bash
mkdir -p /root/VEC_mig_caching
cd /root/VEC_mig_caching
```

#### 步骤3: 上传项目文件

**选项1 - 使用Git（推荐，如果项目在GitHub）**:
```bash
# 在服务器上
git clone <你的项目地址> /root/VEC_mig_caching
```

**选项2 - 使用SCP（从本地上传）**:
```bash
# 在本地Windows PowerShell执行
# 先压缩项目（排除大文件）
Compress-Archive -Path D:\VEC_mig_caching\* -DestinationPath D:\VEC_project.zip

# 上传到服务器
scp -P 19287 D:\VEC_project.zip root@region-9.autodl.pro:/root/

# 在服务器上解压
ssh -p 19287 root@region-9.autodl.pro
cd /root
unzip VEC_project.zip -d VEC_mig_caching
```

**选项3 - 使用WinSCP（图形化工具）**:
1. 下载WinSCP: https://winscp.net/
2. 新建连接:
   - 主机名: region-9.autodl.pro
   - 端口: 19287
   - 用户名: root
   - 密码: dfUJkmli0mHk
3. 拖拽上传整个项目文件夹

#### 步骤4: 配置服务器环境
```bash
cd /root/VEC_mig_caching

# 检查Python和CUDA
python --version
nvcc --version
nvidia-smi

# 安装依赖（使用清华镜像加速）
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple

# 验证PyTorch和CUDA
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

---

## 🏃 开始训练

### 方法1: 使用训练脚本（后台运行，推荐）

在服务器上运行：
```bash
cd /root/VEC_mig_caching

# 启动TD3训练（200轮）
nohup python train_single_agent.py --algorithm TD3 --episodes 200 > training_td3.log 2>&1 &

# 查看日志
tail -f training_td3.log

# 或使用部署脚本创建的便捷脚本
./start_training.sh TD3 200
```

### 方法2: 前台运行（简单测试）
```bash
# 快速测试（30轮）
python train_single_agent.py --algorithm TD3 --episodes 30

# 完整训练（200轮）
python train_single_agent.py --algorithm TD3 --episodes 200
```

### 方法3: 使用screen保持会话
```bash
# 安装screen（如果未安装）
apt install screen

# 创建新会话
screen -S vec_training

# 启动训练
python train_single_agent.py --algorithm TD3 --episodes 200

# 按 Ctrl+A 然后按 D 离开会话
# 重新连接: screen -r vec_training
```

---

## 📊 监控训练

### 查看训练进度
```bash
# 实时查看日志
tail -f training_td3.log

# 查看GPU使用
nvidia-smi

# 查看进程
ps aux | grep train_single_agent

# 查看结果目录
ls -lh results/single_agent/td3/
```

### 使用监控脚本（如果用了自动部署）
```bash
./monitor_training.sh
```

---

## 💾 下载训练结果

### 从服务器下载到本地

**Linux/Mac**:
```bash
# 下载整个results目录
scp -P 19287 -r root@region-9.autodl.pro:/root/VEC_mig_caching/results ./results_from_server

# 下载单个算法结果
scp -P 19287 -r root@region-9.autodl.pro:/root/VEC_mig_caching/results/single_agent/td3 ./td3_results
```

**Windows PowerShell**:
```powershell
# 下载结果
scp -P 19287 -r root@region-9.autodl.pro:/root/VEC_mig_caching/results D:\results_from_server

# 或使用WinSCP图形化下载
```

---

## 🎓 训练任务建议

### 快速验证（约30分钟）
```bash
python train_single_agent.py --algorithm TD3 --episodes 30
```

### 标准训练（约2-3小时）
```bash
python train_single_agent.py --algorithm TD3 --episodes 200
```

### 学术完整实验（约6-8小时）
```bash
# Baseline对比
python run_academic_experiments.py --mode baseline --episodes 200

# 消融实验
python run_academic_experiments.py --mode ablation --episodes 200
```

### 多算法对比（约8-10小时）
```bash
python train_single_agent.py --compare --episodes 200
```

---

## 🔧 常见问题

### 1. GPU不可用
```bash
# 检查CUDA
nvidia-smi
nvcc --version

# 重新安装PyTorch（CUDA 11.8版本）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 2. 训练被中断
```bash
# 使用nohup防止SSH断开时终止
nohup python train_single_agent.py --algorithm TD3 --episodes 200 > training.log 2>&1 &

# 或使用screen保持会话
screen -S training
python train_single_agent.py --algorithm TD3 --episodes 200
# Ctrl+A then D to detach
```

### 3. 内存不足
```bash
# 减少batch size
# 编辑 config/system_config.py
# config.training.batch_size = 64  # 改小一点

# 或使用梯度累积（在train_single_agent.py中添加）
```

### 4. 停止训练
```bash
# 查找进程ID
ps aux | grep train_single_agent

# 停止进程
kill <PID>

# 强制停止
kill -9 <PID>
```

---

## 📈 性能优化建议

### AutoDL平台优化
```bash
# 1. 使用镜像源加速
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 2. 设置PyTorch优化
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# 3. 监控资源使用
watch -n 1 nvidia-smi
```

### 训练加速技巧
```python
# 在config/system_config.py中调整：
config.training.batch_size = 256      # 增大batch size（如果GPU内存够）
config.training.num_workers = 4       # 使用多线程加载数据
config.experiment.save_frequency = 50 # 减少保存频率
```

---

## 🎯 完整工作流示例

```bash
# 1. 连接服务器
ssh -p 19287 root@region-9.autodl.pro

# 2. 进入项目
cd /root/VEC_mig_caching

# 3. 快速测试
python train_single_agent.py --algorithm TD3 --episodes 30

# 4. 确认无误后，启动完整训练（后台）
nohup python train_single_agent.py --algorithm TD3 --episodes 200 > training_td3.log 2>&1 &

# 5. 记录进程ID
echo $! > training.pid

# 6. 断开连接（训练继续）
exit

# 7. 稍后重新连接检查
ssh -p 19287 root@region-9.autodl.pro
cd /root/VEC_mig_caching
tail -f training_td3.log

# 8. 训练完成后下载结果（在本地执行）
scp -P 19287 -r root@region-9.autodl.pro:/root/VEC_mig_caching/results ./results_from_server
```

---

## 📞 需要帮助？

如果遇到问题：
1. 检查日志文件
2. 运行 `./monitor_training.sh`（如果用了自动部署）
3. 查看GPU状态：`nvidia-smi`
4. 检查Python环境：`python -c "import torch; print(torch.__version__, torch.cuda.is_available())"`

---

**部署时间**: 2025-10-29  
**服务器类型**: AutoDL GPU服务器  
**推荐GPU**: RTX 3090 / A100 / V100

