# AutoDL 带宽成本对比实验部署指南

## 📋 实验概述

**实验目标**: 在AutoDL服务器上运行RSU计算资源对比实验  
**服务器信息**: 
- 主机: region-41.seetacloud.com
- 端口: 38597
- 用户名: root
- 密码: dXI7ldI+vPec

**实验参数**:
- 实验类型: `rsu_compute` (RSU计算资源敏感性分析)
- 计算资源档位: 默认5档 (30.0, 40.0, 50.0, 60.0, 70.0 GHz)
- 训练轮次: 1200 (TD3策略), 300 (启发式策略)
- 随机种子: 42

---

## 🚀 快速开始

### 方法1: Linux/macOS/Git Bash (推荐)

```bash
# 1. 进入部署目录
cd d:/VEC_mig_caching/deployment

# 2. 赋予执行权限
chmod +x autodl_deploy_bandwidth_experiment.sh

# 3. 运行部署脚本
./autodl_deploy_bandwidth_experiment.sh
```

### 方法2: Windows PowerShell/CMD

```batch
# 1. 进入部署目录
cd d:\VEC_mig_caching\deployment

# 2. 运行部署脚本
autodl_deploy_bandwidth_experiment.bat
```

**注意**: Windows版本需要安装 [PuTTY工具套件](https://www.putty.org/)，建议使用Git Bash运行Linux脚本。

---

## 📖 部署流程说明

脚本会自动执行以下6个步骤：

### [1/6] 测试服务器连接
- 验证SSH连接是否正常
- 确认密码和端口配置

### [2/6] 创建远程目录
- 在服务器上创建 `/root/VEC_mig_caching` 目录

### [3/6] 同步项目文件
- 使用rsync上传项目代码
- 自动排除不必要的文件 (`__pycache__`, `results/`, `.git/`等)
- 预计时间: 3-10分钟 (取决于网络速度)

### [4/6] 配置服务器环境
- 检查Python、CUDA、GPU环境
- 安装项目依赖 (使用清华镜像加速)
- 验证PyTorch和CUDA是否正常

### [5/6] 创建训练脚本
自动生成以下管理脚本:
- `start_bandwidth_experiment.sh`: 启动实验
- `monitor_experiment.sh`: 监控实验状态
- `stop_experiment.sh`: 停止实验

### [6/6] 启动实验
- 在后台启动实验
- 输出日志到 `bandwidth_experiment_<timestamp>.log`
- 保存进程ID到 `bandwidth_experiment.pid`

---

## 🔍 监控实验进度

### 1. 连接到服务器

```bash
ssh -p 38597 root@region-41.seetacloud.com
# 输入密码: dXI7ldI+vPec
```

### 2. 进入项目目录

```bash
cd /root/VEC_mig_caching
```

### 3. 查看实验状态

```bash
# 快速查看状态
./monitor_experiment.sh

# 实时查看日志
tail -f bandwidth_experiment_*.log

# 监控GPU使用情况
watch -n 5 nvidia-smi

# 查看进程
ps aux | grep python
```

### 4. 关键日志标识

实验正常运行时，日志应包含:
```
>>> Running RSU total compute sensitivity experiment (GHz)
[1/5] 30.0 GHz
  Strategy: local-only
    Episode 100/1200 | Cost: xxx
```

---

## ⏱️ 预计实验时间

| 策略类型 | 训练轮次 | 单配置时间 | 5配置总时间 |
|---------|---------|-----------|-----------|
| TD3策略 (2个) | 1200轮 | ~5-6小时 | ~25-30小时 |
| 启发式策略 (4个) | 300轮 | ~1-1.5小时 | ~5-6小时 |
| **总计** | - | - | **~30-38小时** |

**注意**: 
- 以上为使用GPU加速的预估时间
- 建议AutoDL实例购买至少40小时运行时长
- 实验在后台运行，可以断开SSH连接

---

## 🛑 停止实验

### 方法1: 使用停止脚本

```bash
cd /root/VEC_mig_caching
./stop_experiment.sh
```

### 方法2: 手动停止

```bash
# 查看进程ID
cat bandwidth_experiment.pid

# 停止进程
kill <PID>

# 或强制停止
kill -9 <PID>
```

---

## 📥 下载实验结果

### 从本地Windows下载 (使用pscp)

```batch
pscp -P 38597 -pw dXI7ldI+vPec -r root@region-41.seetacloud.com:/root/VEC_mig_caching/results ./results_from_autodl
```

### 从本地Linux/macOS下载 (使用scp)

```bash
scp -P 38597 -r root@region-41.seetacloud.com:/root/VEC_mig_caching/results ./results_from_autodl
```

### 结果目录结构

```
results/parameter_sensitivity/
└── bandwidth_<timestamp>/
    └── rsu_compute/
        ├── summary.json                    # 实验总结
        ├── rsu_compute_vs_total_cost.png   # 成本对比图
        ├── rsu_compute_vs_delay.png        # 时延对比图
        ├── rsu_compute_vs_normalized_cost.png
        ├── rsu_compute_vs_throughput.png
        ├── rsu_compute_vs_rsu_utilization.png
        └── ... (其他性能指标图表)
```

---

## ⚠️ 常见问题

### 1. 连接失败

**问题**: `sshpass: command not found` 或 `Connection refused`

**解决方案**:
```bash
# Linux/WSL安装sshpass
sudo apt install sshpass

# macOS安装sshpass
brew install hudochenkov/sshpass/sshpass

# 或手动连接后执行命令
ssh -p 38597 root@region-41.seetacloud.com
cd /root/VEC_mig_caching
./start_bandwidth_experiment.sh
```

### 2. 依赖安装失败

**问题**: `pip install` 超时或失败

**解决方案**:
```bash
# 在服务器上手动安装
ssh -p 38597 root@region-41.seetacloud.com
cd /root/VEC_mig_caching
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. GPU不可用

**问题**: `CUDA可用: False`

**解决方案**:
```bash
# 检查GPU
nvidia-smi

# 检查CUDA版本
nvcc --version

# 检查PyTorch是否支持当前CUDA版本
python -c "import torch; print(torch.version.cuda)"
```

### 4. 实验进程意外停止

**问题**: 日志显示实验中断

**可能原因**:
- AutoDL实例时长耗尽（自动关机）
- 内存不足 (OOM)
- 磁盘空间不足

**解决方案**:
```bash
# 检查磁盘空间
df -h

# 检查内存使用
free -h

# 重新启动实验
./start_bandwidth_experiment.sh
```

---

## 🔧 手动执行命令

如果自动化脚本失败，可以手动执行以下命令：

### 1. 手动上传代码

```bash
# 使用git (推荐)
ssh -p 38597 root@region-41.seetacloud.com
cd /root
git clone <your-repo-url> VEC_mig_caching
cd VEC_mig_caching

# 或使用rsync
rsync -avz -e "ssh -p 38597" \
    --exclude '__pycache__' --exclude 'results/' \
    ./ root@region-41.seetacloud.com:/root/VEC_mig_caching/
```

### 2. 手动配置环境

```bash
ssh -p 38597 root@region-41.seetacloud.com
cd /root/VEC_mig_caching
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 手动启动实验

```bash
# 前台运行（用于调试）
python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py \
    --experiment-types rsu_compute \
    --rsu-compute-levels default \
    --episodes 1200 \
    --seed 42 \
    --optimize-heuristic

# 后台运行
nohup python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py \
    --experiment-types rsu_compute \
    --rsu-compute-levels default \
    --episodes 1200 \
    --seed 42 \
    --optimize-heuristic \
    > bandwidth_experiment.log 2>&1 &
```

---

## 📊 实验完成后检查

### 1. 验证结果完整性

```bash
cd /root/VEC_mig_caching/results/parameter_sensitivity
ls -la bandwidth_*/rsu_compute/

# 应该包含:
# - summary.json (实验总结)
# - 8个以上的PNG图表文件
# - 各策略的训练日志
```

### 2. 快速查看结果

```bash
# 查看summary.json
cat results/parameter_sensitivity/bandwidth_*/rsu_compute/summary.json | python -m json.tool

# 检查图表数量
ls -l results/parameter_sensitivity/bandwidth_*/rsu_compute/*.png | wc -l
```

---

## 💡 优化建议

### 1. 使用快速模式 (快速验证)

```bash
python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py \
    --experiment-types rsu_compute \
    --rsu-compute-levels "30.0,50.0,70.0" \  # 仅3个配置点
    --episodes 500 \                         # 减少轮次
    --seed 42 \
    --optimize-heuristic \
    --fast-mode
```

预计时间: ~10-12小时 (节省约67%时间)

### 2. 仅运行特定策略

```bash
python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py \
    --experiment-types rsu_compute \
    --strategies comprehensive-migration,local-only,remote-only \
    --episodes 1200 \
    --seed 42
```

---

## 📞 技术支持

如有问题，请检查：
1. AutoDL控制台的实例状态
2. 服务器日志: `bandwidth_experiment_*.log`
3. GPU监控: `nvidia-smi`
4. 磁盘空间: `df -h`

---

## 📝 附录

### 完整命令参数说明

```
python experiments/td3_strategy_suite/run_bandwidth_cost_comparison.py \
    --experiment-types rsu_compute \        # 实验类型: bandwidth/rsu_compute/uav_compute
    --rsu-compute-levels default \          # RSU计算资源档位 (default=5档)
    --episodes 1200 \                       # TD3训练轮次
    --seed 42 \                             # 随机种子
    --optimize-heuristic \                  # 优化启发式策略 (使用300轮)
    --central-resource \                    # 使用中心化资源管理
    --strategies <list> \                   # 指定策略 (可选)
    --fast-mode                             # 快速验证模式 (可选)
```

### 策略列表

- `comprehensive-migration`: CAMTD3 (完整策略+迁移)
- `comprehensive-no-migration`: TD3 (完整策略无迁移)
- `local-only`: 仅本地执行
- `remote-only`: 仅远程执行
- `offloading-only`: 仅卸载决策
- `resource-only`: 仅资源分配

---

**最后更新**: 2025-11-19  
**版本**: 1.0
